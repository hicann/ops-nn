// ----------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_rotate_quant.h"

namespace {
struct AclTensorDeleter {
    void operator()(aclTensor* t) const
    {
        if (t != nullptr) {
            aclDestroyTensor(t);
        }
    }
};
using AclTensorPtr = std::unique_ptr<aclTensor, AclTensorDeleter>;

AclTensorPtr WrapTensor(const at::Tensor& t, aclDataType dt)
{
    std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
    return AclTensorPtr(aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                        shape.size(), const_cast<void*>(t.data_ptr())));
}

void* EnsureWs(uint64_t need, void*& buf, uint64_t& cap)
{
    if (need > cap) {
        if (buf != nullptr) {
            aclrtFree(buf);
        }
        auto ret = aclrtMalloc(&buf, need, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMalloc failed, ret=", ret);
        cap = need;
    }
    return need ? buf : nullptr;
}

int64_t RotateQuant(const at::Tensor& x, const at::Tensor& rot, const at::Tensor& y, const at::Tensor& scale,
                    int64_t streamPtr)
{
    TORCH_CHECK(x.dim() == 2, "rotate baseline expects x shape (M, N)");
    TORCH_CHECK(rot.dim() == 2, "rotate baseline expects rotation shape (N, N)");
    TORCH_CHECK(y.sizes() == x.sizes(), "rotate baseline expects y shape equal to x shape");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == x.size(0), "rotate baseline expects scale shape (M)");
    TORCH_CHECK(x.scalar_type() == at::kHalf && rot.scalar_type() == at::kHalf,
                "rotate baseline only wraps FP16 input and rotation");
    TORCH_CHECK(y.scalar_type() == at::kChar, "rotate baseline expects INT8 y");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "rotate baseline expects FP32 scale");
    TORCH_CHECK(x.is_contiguous() && rot.is_contiguous() && y.is_contiguous() && scale.is_contiguous(),
                "rotate baseline expects contiguous tensors to avoid hidden copies");

    auto stream = reinterpret_cast<aclrtStream>(streamPtr);
    auto xt = WrapTensor(x, ACL_FLOAT16);
    auto rt = WrapTensor(rot, ACL_FLOAT16);
    auto yt = WrapTensor(y, ACL_INT8);
    auto st = WrapTensor(scale, ACL_FLOAT);
    char roundMode[] = "rint";
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnRotateQuantGetWorkspaceSize(xt.get(), rt.get(), nullptr, -1, roundMode, 0, 0.0, false, yt.get(),
                                                st.get(), &ws, &exec);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnRotateQuantGetWorkspaceSize failed, ret=", ret);
    static void* buf = nullptr;
    static uint64_t cap = 0;
    ret = aclnnRotateQuant(EnsureWs(ws, buf, cap), ws, exec, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnRotateQuant failed, ret=", ret);
    return 0;
}
} // namespace

TORCH_LIBRARY(rotate_baseline, m)
{
    m.def("rotate_quant(Tensor x, Tensor rotation, Tensor y, Tensor scale, int stream) -> int");
}

TORCH_LIBRARY_IMPL(rotate_baseline, PrivateUse1, m) { m.impl("rotate_quant", &RotateQuant); }
