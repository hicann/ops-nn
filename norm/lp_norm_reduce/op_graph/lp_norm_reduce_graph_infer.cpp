/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// LpNormReduce_package/op_graph/lp_norm_reduce_graph_infer.cpp
// =============================================================================
//
// ROLE: Graph-level data type inference for the LpNormReduce operator.
//   本文件为 GEIR 图模式数据类型推导实现：
//   - 注册链 IMPL_OP(LpNormReduce).InferDataType(InferDataTypeForLpNormReduce)
//     原样保留（签名不变：gert::InferDataTypeContext* → ge::graphStatus）；
//   - 推导规则按 docs/LpNormReduce/design/InferShapeDtype.md「InferDataType」节：
//     y.dtype = x.dtype（promotion = same_as_first_input，无跨 dtype 组合）。
//   - 非法 dtype 校验：与公共原型和 canndev 原生 runtime InferDataType
//     （op_proto/runtime/lp_norm.cc InferDataType4LpNorm）一致，输入须为
//     DT_FLOAT16 / DT_FLOAT / DT_BF16，越界报 dtype_not_supported。
//
// CONTENTS:
//   - InferDataTypeForLpNormReduce() — the type inference function（真实实现）
//   - IMPL_OP(LpNormReduce).InferDataType(...) — registration macro
//
// =============================================================================

#include <string>

#include "register/op_impl_registry.h"                // IMPL_OP macro for operator registration
#include "exe_graph/runtime/infer_datatype_context.h" // InferDataTypeContext
#include "graph/utils/type_utils.h"                   // ge::TypeUtils::DataTypeToSerialString
#include "op_common/log/log.h"                        // OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON

using namespace ge;

namespace ops {

// ---------------------------------------------------------------------------
// InferDataTypeForLpNormReduce(context) — data type inference callback
//
// y.dtype = x.dtype（唯一输入决定唯一输出，无 dtype promotion；fp16 输入必
// fp16 输出、fp32 输入必 fp32 输出、bf16 输入必 bf16 输出）。输入 dtype 不在
// 950 交付的三档（DT_FLOAT16 / DT_FLOAT / DT_BF16）内 → 推导失败。
// ---------------------------------------------------------------------------
static ge::graphStatus InferDataTypeForLpNormReduce(gert::InferDataTypeContext* context)
{
    const ge::DataType xDtype = context->GetInputDataType(0);
    OP_CHECK_IF(xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_FLOAT && xDtype != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(xDtype),
                                                      "only DT_FLOAT16/DT_FLOAT/DT_BF16 are supported on ascend950"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, xDtype);
    return ge::GRAPH_SUCCESS;
}

// IMPL_OP(LpNormReduce).InferDataType(func):
//   Registers InferDataTypeForLpNormReduce as the type inference function
//   for the LpNormReduce operator type（注册链原样保留）。
IMPL_OP(LpNormReduce).InferDataType(InferDataTypeForLpNormReduce);
} // namespace ops
