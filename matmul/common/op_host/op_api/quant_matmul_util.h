/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_QUANT_MATMUL_UTIL_H
#define OPS_NN_QUANT_MATMUL_UTIL_H

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

namespace Ops {
namespace NN {

inline const aclTensor* ContiguousMxScale(const aclTensor* scale, aclOpExecutor* executor)
{
    CHECK_RET(scale != nullptr && executor != nullptr, nullptr);
    if (scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0 || op::IsContiguous(scale)) {
        return l0op::Contiguous(scale, executor);
    }

    // Contiguous may dispatch to kernels without E8M0 support. E8M0 and UINT8 are both
    // one byte: reinterpret the descriptor to copy the encoding without a numeric cast.
    // Build a UINT8 view backed by the original storage, then restore the logical
    // shape, strides and offset. Contiguous copies the logical view elements,
    // applying the view offset exactly once.
    auto byteView = executor->CreateView(scale, scale->GetStorageShape(), 0);
    CHECK_RET(byteView != nullptr, nullptr);
    byteView->SetViewShape(scale->GetViewShape());
    byteView->SetViewStrides(scale->GetViewStrides());
    byteView->SetViewOffset(scale->GetViewOffset());
    byteView->SetViewFormat(scale->GetViewFormat());
    byteView->SetStorageFormat(scale->GetStorageFormat());
    byteView->SetOriginalFormat(scale->GetOriginalFormat());
    byteView->SetOriginalShape(scale->GetViewShape());
    byteView->SetDataType(op::DataType::DT_UINT8);

    const auto contiguousBytes = l0op::Contiguous(byteView, executor);
    CHECK_RET(contiguousBytes != nullptr, nullptr);
    // Keep the descriptor captured by the copy task as UINT8; restore E8M0 on a new view.
    auto result = executor->CreateView(contiguousBytes, contiguousBytes->GetViewShape(),
                                       contiguousBytes->GetViewOffset());
    CHECK_RET(result != nullptr, nullptr);
    result->SetDataType(op::DataType::DT_FLOAT8_E8M0);
    return result;
}

} // namespace NN
} // namespace Ops

#endif // OPS_NN_QUANT_MATMUL_UTIL_H
