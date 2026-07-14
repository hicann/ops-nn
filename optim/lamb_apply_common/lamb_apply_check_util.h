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
 * \file lamb_apply_check_util.h
 * \brief lamb_apply_optimizer_assign / lamb_apply_weight_assign 复用的输入校验(dtype 一致性、标量非空)。
 */
#ifndef OPS_OPTIM_LAMB_APPLY_COMMON_CHECK_UTIL_H
#define OPS_OPTIM_LAMB_APPLY_COMMON_CHECK_UTIL_H

#include <cstddef>
#include <string>
#include "exe_graph/runtime/tiling_context.h"
#include "log/log.h"

namespace optiling {

// 所有输入(1..inputNum-1)、输出(0..outputNum-1)的 dtype 必须与 input0 一致。
inline ge::graphStatus CheckLambApplyDtypeConsistency(gert::TilingContext* context, int32_t inputNum,
                                                      const char* const inputNames[], int32_t outputNum,
                                                      const char* const outputNames[])
{
    auto input0Desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    for (int32_t inputIdx = 1; inputIdx < inputNum; inputIdx++) {
        auto inputDesc = context->GetInputDesc(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
        if (inputDesc->GetDataType() != input0DType) {
            std::string paramNames = std::string(inputNames[inputIdx]) + " and input0";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), paramNames.c_str(),
                (Ops::Base::ToString(inputDesc->GetDataType()) + " and " + Ops::Base::ToString(input0DType)).c_str(),
                "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    for (int32_t outputIdx = 0; outputIdx < outputNum; outputIdx++) {
        auto outputDesc = context->GetOutputDesc(outputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
        if (outputDesc->GetDataType() != input0DType) {
            std::string paramNames = std::string(outputNames[outputIdx]) + " and input0";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                context->GetNodeName(), paramNames.c_str(),
                (Ops::Base::ToString(outputDesc->GetDataType()) + " and " + Ops::Base::ToString(input0DType)).c_str(),
                "Their dtypes should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 标量类输入(系数)为每元素计算所必需, 空Tensor 视为缺失必选值(畸形输入), 不支持。
inline ge::graphStatus CheckLambApplyScalarNotEmpty(gert::TilingContext* context, const int32_t* scalarIdx,
                                                    size_t scalarCount, const char* const inputNames[])
{
    for (size_t i = 0; i < scalarCount; i++) {
        int32_t idx = scalarIdx[i];
        auto scalarShape = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, scalarShape);
        if (scalarShape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), inputNames[idx],
                                                  Ops::Base::ToString(scalarShape->GetStorageShape()).c_str(),
                                                  "scalar input does not support empty tensor");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

#endif // OPS_OPTIM_LAMB_APPLY_COMMON_CHECK_UTIL_H
