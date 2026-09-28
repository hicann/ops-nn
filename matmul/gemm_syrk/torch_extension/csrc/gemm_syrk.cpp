/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// GemmSyrk PTA (PyTorch Adapter) C++ 后端
// 把 PyTorch at::Tensor 桥接到 aclnnGemmSyrk(由算子本体 op_api/aclnn_gemm_syrk.cpp 提供)。
//
// 与 README 模板的差异: 不 #include "aclnnop/aclnn_gemm_syrk.h"，也不直接使用 ACLNN_CMD 宏传 at::Tensor。
//   1) ACLNN_CMD 宏用 #aclnn_api 字符串化 + dlsym 运行时解析,不依赖头里的函数声明;
//      且 gemm_syrk 是自定义算子,头装在 vendor 包(<vendor>/op_api/include)而非 cann 标准 include。
//   2) aclnn_common.h 的 ConvertType(at::Tensor) 对 3D 张量自动映射 ACL_FORMAT_NCL
//      (PrepareTensorMeta 的 dimNum 分支),而 aclnnGemmSyrk 的 CheckParams 仅接受 ND,
//      3D (batch, m, k) 输入会报 "a only supports ND format"。因此这里手工构造
//      ACL_FORMAT_ND 的 aclTensor,再按 ACLNN_CMD 相同的两段式流程调用。
//   运行期: 算子包安装后,libcust_opapi.so 提供 aclnnGemmSyrk 符号(dlsym 解析)。

#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

#include "aclnn_common.h"

namespace cann_ops_nn {
namespace matmul {
namespace {

constexpr int64_t MIN_DIM_NUM = 2;
constexpr int64_t MAX_DIM_NUM = 6;

aclScalar* CreateOptionalFloatScalar(const c10::optional<double>& value)
{
    if (!value.has_value()) {
        return nullptr; // aclnnGemmSyrk: nullptr 时默认 1.0
    }
    static const auto aclCreateScalar = GET_OP_API_FUNC(CreateScalar);
    TORCH_CHECK(aclCreateScalar != nullptr, "aclCreateScalar is not available");
    float scalarValue = static_cast<float>(value.value());
    return aclCreateScalar(&scalarValue, ACL_FLOAT);
}

// 构造 ACL_FORMAT_ND 的 aclTensor（绕开 ConvertType 对 3D 的 NCL 自动映射）。
aclTensor* CreateAclTensorND(const at::Tensor& tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(CreateTensor);
    TORCH_CHECK(aclCreateTensor != nullptr, "aclCreateTensor is not available");
    const aclDataType aclDataTypeValue = ConvertToAclDataType(tensor.scalar_type());
    const auto sizes = tensor.sizes();
    const auto strides = tensor.strides();
    std::vector<int64_t> storageDims;
    storageDims.push_back(tensor.storage().nbytes() / tensor.itemsize());
    return aclCreateTensor(sizes.data(), sizes.size(), aclDataTypeValue, strides.data(),
                           ConvertToAclStorageOffset(tensor, aclDataTypeValue), ACL_FORMAT_ND, storageDims.data(),
                           storageDims.size(), const_cast<void*>(tensor.storage().data()));
}

void CheckGemmSyrkInputs(const at::Tensor& a, const at::Tensor& c, bool transposeX)
{
    TORCH_CHECK(a.device().type() == at::kPrivateUse1, "gemm_syrk: a must be on NPU device");
    TORCH_CHECK(c.device().type() == at::kPrivateUse1, "gemm_syrk: c must be on NPU device");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
                "gemm_syrk: a dtype must be float16 or bfloat16, got ", a.scalar_type());
    TORCH_CHECK(a.scalar_type() == c.scalar_type(), "gemm_syrk: a and c must have the same dtype");
    TORCH_CHECK(a.dim() >= MIN_DIM_NUM && a.dim() <= MAX_DIM_NUM,
                "gemm_syrk: a must be 2D..6D (..., m, k) or the transposed (..., k, m), got ", a.dim(), " dims");
    TORCH_CHECK(a.dim() == c.dim(), "gemm_syrk: a and c must have the same number of dims");
    TORCH_CHECK(c.size(-1) == c.size(-2), "gemm_syrk: c must be a square matrix (m, m) or (batch, m, m)");
    // transpose_x = false: a 是 [..., m, k]，m 为倒数第二维；
    // transpose_x = true: a 是转置的 [..., k, m] 存储，m 为最后一维。
    const int64_t m = transposeX ? a.size(-1) : a.size(-2);
    TORCH_CHECK(c.size(-2) == m, "gemm_syrk: m axis of a and c must match (m = a.size(-1) when transpose_x)");
    for (int64_t i = 0; i + MIN_DIM_NUM < a.dim(); ++i) {
        TORCH_CHECK(a.size(i) == c.size(i), "gemm_syrk: batch axis ", i, " of a and c must match (no broadcast)");
    }
    TORCH_CHECK(a.is_contiguous() && c.is_contiguous(), "gemm_syrk: a and c must be contiguous");
}

} // namespace

// GemmSyrk: C = alpha * (A @ A^T) + beta * C（transpose_x=True 时 C = alpha * (A^T @ A) + beta * C），
// 原地写回 c 并返回 c。a: (m, k)/(batch, m, k) 或转置 (k, m)/(batch, k, m)，fp16/bf16；
// c: (m, m)/(batch, m, m) 对称矩阵（输入输出同地址）。
at::Tensor gemm_syrk(const at::Tensor& a, const at::Tensor& c, const c10::optional<double>& alpha,
                     const c10::optional<double>& beta, bool transposeX, const std::string& fillMode)
{
    CheckGemmSyrkInputs(a, c, transposeX);

    const c10::OptionalDeviceGuard device_guard(a.device());
    static const auto getWorkspaceSizeAddr = GetOpApiFuncAddr("aclnnGemmSyrkGetWorkspaceSize");
    static const auto opApiAddr = GetOpApiFuncAddr("aclnnGemmSyrk");
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
    TORCH_CHECK(getWorkspaceSizeAddr != nullptr && opApiAddr != nullptr,
                "aclnnGemmSyrk or aclnnGemmSyrkGetWorkspaceSize not found in ", GetCustOpApiLibName(), " or ",
                GetNnOpApiLibName(), ".");

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    if (c10_npu::check_enqueue_need_use(aclStream)) {
        aclrtUseStreamResInCurrentThread(aclStream);
    }

    aclTensor* aclA = CreateAclTensorND(a);
    aclTensor* aclC = CreateAclTensorND(c);
    aclScalar* alphaScalar = CreateOptionalFloatScalar(alpha);
    aclScalar* betaScalar = CreateOptionalFloatScalar(beta);
    const char* fillModeCStr = fillMode.c_str();

    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc != nullptr) {
        initMemFunc(nullptr, false);
    }
    ApplyDeterministicConfig();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    using GetWorkspaceSizeFunc = int (*)(aclTensor*, aclTensor*, aclScalar*, aclScalar*, bool, const char*, uint64_t*,
                                         aclOpExecutor**);
    auto getWorkspaceSizeFunc = reinterpret_cast<GetWorkspaceSizeFunc>(getWorkspaceSizeAddr);
    auto workspaceStatus = getWorkspaceSizeFunc(aclA, aclC, alphaScalar, betaScalar, transposeX, fillModeCStr,
                                                &workspaceSize, &executor);
    if (workspaceStatus != 0) {
        Release(aclA);
        Release(aclC);
        Release(alphaScalar);
        Release(betaScalar);
        if (unInitMemFunc != nullptr) {
            unInitMemFunc(nullptr, false);
        }
        TORCH_CHECK(false, "call aclnnGemmSyrkGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());
    }

    void* workspaceAddr = nullptr;
    at::Tensor workspaceTensor;
    if (workspaceSize != 0) {
        at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
        workspaceTensor = at::empty({static_cast<int64_t>(workspaceSize)}, options.dtype(at::kByte));
        workspaceAddr = const_cast<void*>(workspaceTensor.storage().data());
    }

    auto aclCall = [aclA, aclC, alphaScalar, betaScalar, workspaceAddr, workspaceSize, executor, aclStream]() -> int {
        if (c10_npu::check_enqueue_need_use(aclStream)) {
            aclrtUseStreamResInCurrentThread(aclStream);
        }
        using OpApiFunc = int (*)(void*, uint64_t, aclOpExecutor*, const aclrtStream);
        auto opApiFunc = reinterpret_cast<OpApiFunc>(opApiAddr);
        auto apiRet = opApiFunc(workspaceAddr, workspaceSize, executor, aclStream);
        Release(aclA);
        Release(aclC);
        Release(alphaScalar);
        Release(betaScalar);
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
        if (releaseMemFunc != nullptr) {
            releaseMemFunc(nullptr, false);
        }
        TORCH_CHECK(apiRet == 0, "call aclnnGemmSyrk failed, detail:", aclGetRecentErrMsg());
        return apiRet;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnGemmSyrk");
    cmd.SetCustomHandler(aclCall);
    cmd.Run();
    if (unInitMemFunc != nullptr) {
        unInitMemFunc(nullptr, false);
    }
    return c;
}

} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_syrk", &cann_ops_nn::matmul::gemm_syrk,
          "GemmSyrk (C = alpha * A @ A^T + beta * C, in-place on c) on NPU");
}
