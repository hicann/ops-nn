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
 * \file bn3_d_training_reduce_grad_graph_infer.cpp
 * \brief BN3DTrainingReduceGrad 的图级数据类型推导（InferDataType）
 */

#include "register/op_impl_registry.h" // IMPL_OP macro for operator registration

using namespace ge;

namespace ops {

// ---------------------------------------------------------------------------
// InferDataTypeForBN3DTrainingReduceGrad(context) — data type inference callback
//
// 校验点（任一失败返回 GRAPH_FAILED，不推导输出 dtype）：
//   1. grads dtype 白名单：{float16, float32, bfloat16}；
//   2. grads 与 x dtype 一致；
//   3. diff_scale / diff_offset / scale / batch_mean / batch_variance
//      必须全部 float32；
//   4. y.dtype = grads.dtype（spec outputs.y.dtype_rule，无 promotion）。
// 中间升位 f32 计算是 kernel 内部口径，不改变输出 dtype。
//
// Parameters:
//   context — InferDataTypeContext that provides access to input dtypes
//             and allows setting output dtypes
//
// Returns:
//   ge::graphStatus — ge::GRAPH_SUCCESS on success
// ---------------------------------------------------------------------------
static ge::graphStatus InferDataTypeForBN3DTrainingReduceGrad(gert::InferDataTypeContext* context)
{
    // 校验点 1: grads dtype 白名单。
    const ge::DataType grads_dtype = context->GetInputDataType(0);
    if (grads_dtype != ge::DT_FLOAT16 && grads_dtype != ge::DT_FLOAT && grads_dtype != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }

    // 校验点 2: grads 与 x dtype 一致。
    if (context->GetInputDataType(1) != grads_dtype) {
        return ge::GRAPH_FAILED;
    }

    // 校验点 3: 5 个 1D 参数张量必须全部 float32。
    for (size_t i = 2U; i < 7U; ++i) {
        if (context->GetInputDataType(i) != ge::DT_FLOAT) {
            return ge::GRAPH_FAILED;
        }
    }

    // 校验点 4: y.dtype = grads.dtype。
    if (context->SetOutputDataType(0, grads_dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// IMPL_OP(BN3DTrainingReduceGrad).InferDataType(func):
//   Registers InferDataTypeForBN3DTrainingReduceGrad as the type inference function
//   for the BN3DTrainingReduceGrad operator type.
IMPL_OP(BN3DTrainingReduceGrad).InferDataType(InferDataTypeForBN3DTrainingReduceGrad);
} // namespace ops
